package fr.istic.service;

import io.quarkus.panache.common.Page;
import fr.istic.domain.StudentResponse;
import fr.istic.domain.ResponseGroup;
import fr.istic.service.customdto.EntityId;
//import fr.istic.service.customdto.ResponseGroupsIdsDto;
import fr.istic.service.dto.ResponseGroupDTO;
import fr.istic.service.mapper.ResponseGroupMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@ApplicationScoped
@Transactional
public class ResponseGroupService {

    private final Logger log = LoggerFactory.getLogger(ResponseGroupService.class);

    @Inject
    ResponseGroupMapper responseGroupMapper;

    /**
     * Persist or update a responseGroup entity.
     *
     * @param responseGroupDTO the DTO of the entity.
     * @return the persisted or updated DTO.
     */
    @Transactional
    public ResponseGroupDTO persistOrUpdate(ResponseGroupDTO responseGroupDTO) {
        log.debug("Request to save ResponseGroup : {}", responseGroupDTO);
        var responseGroup = responseGroupMapper.toEntity(responseGroupDTO);
        responseGroup = ResponseGroup.persistOrUpdate(responseGroup);
        return responseGroupMapper.toDto(responseGroup);
    }

    /**
     * Delete the ResponseGroup by ID.
     *
     * @param id the ID of the entity.
     */
    @Transactional
    public void delete(Long id) {
        log.debug("Request to delete ResponseGroup : {}", id);
        ResponseGroup.findByIdOptional(id).ifPresent(responseGroup -> {
            // Delete the responseGroup entity
            responseGroup.delete();
        });
    }


        /**
     * Delete the ResponseGroup by ID.
     *
     * @param id the ID of the entity.
     */
    @Transactional
    public void deleteByQuestionId(Long questionId) {
        log.debug("Request to delete ResponseGroup for question  {}", questionId);
        ResponseGroup.deleteByQId(questionId);
    }



    /**
     * Get one ResponseGroup by ID.
     *
     * @param id the ID of the entity.
     * @return the entity as an optional DTO.
     */
    public Optional<ResponseGroupDTO> findOne(Long id) {
        log.debug("Request to get ResponseGroup : {}", id);
        return ResponseGroup.findByIdOptional(id)
            .map(responseGroup -> responseGroupMapper.toDto((ResponseGroup) responseGroup));
    }

    /**
     * Get all the ResponseGroups.
     *
     * @param page the pagination information.
     * @return a paged list of entities.
     */
    public Paged<ResponseGroupDTO> findAll(Page page) {
        log.debug("Request to get all ResponseGroups");
        return new Paged<>(ResponseGroup.findAll().page(page))
            .map(responseGroup -> responseGroupMapper.toDto((ResponseGroup) responseGroup));
    }

    /**
     * Get all the ResponseGroups by Question ID.
     *
     * @param page the pagination information.
     * @param questionId the ID of the related question.
     * @return a paged list of entities.
     */
    public Paged<ResponseGroupDTO> findResponseGroupByQuestionId(Page page, long questionId) {
        log.debug("Request to get all ResponseGroups by Question ID");
        return new Paged<>(ResponseGroup.findByQuestionId(questionId).page(page))
            .map(responseGroup -> responseGroupMapper.toDto((ResponseGroup) responseGroup));
    }

}
