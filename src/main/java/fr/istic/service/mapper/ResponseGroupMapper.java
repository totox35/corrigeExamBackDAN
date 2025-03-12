package fr.istic.service.mapper;

import fr.istic.domain.*;
import fr.istic.service.dto.ResponseGroupDTO;

import org.mapstruct.*;

/**
 * Mapper for the entity {@link ResponseGroup} and its DTO {@link ResponseGroupDTO}.
 */
@Mapper(componentModel = "jakarta", uses = {QuestionMapper.class, ExamSheetMapper.class}, injectionStrategy = InjectionStrategy.CONSTRUCTOR)
public interface ResponseGroupMapper extends EntityMapper<ResponseGroupDTO, ResponseGroup> {

    @Mapping(source = "question.id", target = "questionId")
    ResponseGroupDTO toDto(ResponseGroup responseGroup);

    @Mapping(source = "questionId", target = "question")
    ResponseGroup toEntity(ResponseGroupDTO responseGroupDTO);

    default ResponseGroup fromId(Long id) {
        if (id == null) {
            return null;
        }
        ResponseGroup responseGroup = new ResponseGroup();
        responseGroup.id = id;
        return responseGroup;
    }
}
